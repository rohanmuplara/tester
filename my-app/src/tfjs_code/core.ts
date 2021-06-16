import * as tf from "@tensorflow/tfjs";
import { downloadImages } from "./image_utils";

/**
 * Tensor references are way more efficent because they don't come back from the gpu to the cpu.
 * So, when you stitch graphs together, everything just happens on gpu. You would set returnTensorReferences
 * for debugging.
 * Additionally, if there is only 1 output tensor, tfjs returns it instead of an array of length.
 *
 * One weird thing is tfjs adds a :0 to all input nodes so this code autodoes to input nodes.
 * Tensor outputnames we have to specify for each model. Rohan couldn't figure out a way to get the output names
 * of the names to match the python names. Thus, this overwrites the names. The outputs of the names correspond
 * to the order in model.json file. Sometimes in the model json file you have will have an output labeled output2(a)
 * and another followed by output1(b). Follow the order in the file and not the numbers so output names would be (a,b)
 * and not in file.
 * We also use object map isntead a proper map because this is what tfjs api accepts.
 *
 */
export async function runModel(
  model: tf.GraphModel,
  tensorMap: any,
  tensorOutputNames: string[],
  returnTensorReferences: boolean
) {
  let renamedTensorMap: any = {};
  for (const tensor_name in tensorMap) {
    const new_tensor_name = tensor_name + ":0";
    renamedTensorMap[new_tensor_name] = tensorMap[tensor_name];
  }
  const predictionsTensor = await model.executeAsync(renamedTensorMap);
  if (returnTensorReferences) {
    return constructMap(tensorOutputNames, predictionsTensor);
  } else {
    if (Array.isArray(predictionsTensor)) {
      const promises = predictionsTensor.map((x) => x.array());
      const arrayTensor = await Promise.all(promises);
      tf.dispose(predictionsTensor);
      return constructMap(tensorOutputNames, arrayTensor);
    } else {
      const arrayTensor = predictionsTensor.arraySync();
      tf.dispose(predictionsTensor);
      return constructMap(tensorOutputNames, arrayTensor);
    }
  }
}
export function constructMap(names: string[], arrayValues: any) {
  let output_dict: any = {};
  // if there is only 1 output tensor, tfjs returns it instead of an array of length 1 so can't iterate like below
  if (names.length === 1) {
    output_dict[names[0]] = arrayValues;
  } else {
    for (let i = 0; i < names.length; i++) {
      let name = names[i];
      let arrayValue = arrayValues[i];
      output_dict[name] = arrayValue;
    }
  }
  return output_dict;
}
export async function drawToCanvas(
  tensor: tf.Tensor4D,
  canvases: HTMLCanvasElement[]
) {
  await Promise.all(
    canvases.map(async (canvas, index) => {
      tensor = tf.cast(tensor, "int32");
      let batch_element = tf.squeeze(tensor, [index]) as tf.Tensor3D;
      await tf.browser.toPixels(batch_element, canvas)!;
      tf.dispose(batch_element);
    })
  );
}
/**
 * Assumes tensor is [batch, height, width, n]
 */
export async function downloadTensorAsImage(
  tensor: tf.Tensor4D,
  names: [string]
) {
  let canvas_height = tensor.shape[1];
  let canvas_width = tensor.shape[2];
  const canvases = names.map(() => {
    let canvas = document.createElement("canvas");
    canvas.height = canvas_height;
    canvas.width = canvas_width;
    return canvas;
  });
  await drawToCanvas(tensor, canvases);
  names.map((name, index) => {
    let canvas = canvases[index];
    let fake_link = document.createElement("a");
    fake_link.download = name;
    fake_link.href = canvas.toDataURL();
    fake_link.click();
  });
}
/*
This takes a raw mask and gives it colors. This is noninituive and little hacking of the api. The params is actually the color array and the mask is indicies
as each mask has a class(integer) that corresponds to the collars array. The batch size,width,height is from mask
and the depth is from the colors array. Assumes a batch, height, width, 1.
*/

export function convertMaskToColors(mask: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => {
    let colors = tf.tensor([
      [0, 0, 0],
      [255, 0, 0],
      [32, 173, 10],
      [117, 112, 2],
      [136, 10, 117],
      [34, 16, 169],
      [36, 121, 142],
      [248, 109, 67],
      [242, 124, 242],
      [208, 97, 48],
      [49, 220, 181],
      [216, 210, 239],
      [27, 50, 31],
      [206, 173, 55],
      [127, 98, 97],
      [255, 229, 85],
      [234, 123, 143],
      [122, 52, 124],
      [242, 68, 46],
      [68, 34, 91],
      [34, 236, 26],
      [236, 26, 84],
      [94, 0, 96],
      [63, 35, 198],
      [110, 103, 165],
      [105, 245, 12],
    ]);
    colors = tf.cast(colors, "int32");
    mask = tf.cast(mask, "int32");
    let spliced_mask = tf.squeeze(mask, [-1]);
    return tf.gather(colors, spliced_mask) as tf.Tensor4D;
  });
}

export async function convertMaskUrlToTensor(
  mask_urls: [string]
): Promise<tf.Tensor4D> {
  let mask_tensors = await Promise.all(
    mask_urls.map(async (mask_url) => {
      let mask = await (await downloadImages([mask_url]))[0];
      let mask_tensor = tf.browser.fromPixels(mask, 1);
      return mask_tensor;
    })
  );
  return tf.stack(mask_tensors) as tf.Tensor4D;
}

export async function convertImageUrlToTensor(
  image_urls: [string]
): Promise<tf.Tensor4D> {
  let image_tensors = await Promise.all(
    image_urls.map(async (image_url) => {
      let image = (await downloadImages([image_url]))[0];
      let image_tensor = tf.browser.fromPixels(image, 3);
      return image_tensor as tf.Tensor3D;
    })
  );
  let stacked_tensor = tf.stack(image_tensors) as tf.Tensor4D;
  tf.dispose(image_tensors);
  return stacked_tensor;
}
