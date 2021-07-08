import * as tf from "@tensorflow/tfjs-core";
import * as tfc from "@tensorflow/tfjs-converter";

import { downloadImages, onload2promise } from "./image_utils";
import { NamedTensor4DMap } from "./base_tfjs";
import { Tensor4D } from "@tensorflow/tfjs-core";

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
  model: tfc.GraphModel,
  tensorMap: any,
  tensorOutputNames: string[],
  isNonBranching: boolean
) {
  let renamedTensorMap: any = {};
  for (const tensor_name in tensorMap) {
    const new_tensor_name = tensor_name + ":0";
    renamedTensorMap[new_tensor_name] = tensorMap[tensor_name];
  }
  let predictionsTensor;
  if (isNonBranching) {
    predictionsTensor = model.execute(renamedTensorMap);
  } else {
    predictionsTensor = await model.executeAsync(renamedTensorMap);
  }
  return constructMap(tensorOutputNames, predictionsTensor);
}
export function constructMap(names: string[], arrayValues: any) {
  let outputDict: any = {};
  // if there is only 1 output tensor, tfjs returns it instead of an array of length 1 so can't iterate like below
  if (names.length === 1) {
    outputDict[names[0]] = arrayValues;
  } else {
    for (let i = 0; i < names.length; i++) {
      let name = names[i];
      let arrayValue = arrayValues[i];
      outputDict[name] = arrayValue;
    }
  }
  return outputDict;
}
export async function drawToCanvas(
  tensor: tf.Tensor4D,
  canvases: HTMLCanvasElement[]
) {
  await Promise.all(
    canvases.map(async (canvas, index) => {
      tensor = tf.cast(tensor, "int32");
      let batchElement = tf.squeeze(tensor, [index]) as tf.Tensor3D;
      await tf.browser.toPixels(batchElement, canvas)!;
      tf.dispose(batchElement);
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
  let canvasHeight = tensor.shape[1];
  let canvasWidth = tensor.shape[2];
  const canvases = names.map(() => {
    let canvas = document.createElement("canvas");
    canvas.height = canvasHeight;
    canvas.width = canvasWidth;
    return canvas;
  });
  await drawToCanvas(tensor, canvases);
  names.forEach((name, index) => {
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
  maskUrls: string[]
): Promise<tf.Tensor4D> {
  let maskTensors = await Promise.all(
    maskUrls.map(async (maskUrl) => {
      let mask = (await downloadImages([maskUrl]))[0];
      let maskTensor = tf.browser.fromPixels(mask, 1);
      let maskFloatTensor = tf.cast(maskTensor, "float32");
      tf.dispose(maskTensor);
      return maskFloatTensor;
    })
  );
  let stacked_tensor = tf.stack(maskTensors) as tf.Tensor4D;
  tf.dispose(maskTensors);
  return stacked_tensor;
}

export async function convertImageUrlsToTensor(
  imageUrls: string[]
): Promise<tf.Tensor4D> {
  let imageTensors = await Promise.all(
    imageUrls.map(async (imageIrl) => {
      let image = (await downloadImages([imageIrl]))[0];
      let imageTensor = tf.browser.fromPixels(image, 3);
      let imageFloatTensor = tf.cast(imageTensor, "float32");
      tf.dispose(imageTensor);
      return imageFloatTensor as tf.Tensor3D;
    })
  );
  let stacked_tensor = tf.stack(imageTensors) as tf.Tensor4D;
  tf.dispose(imageTensors);
  return stacked_tensor;
}

export async function convertDataUrlsToTensor(
  dataUrls: string[]
): Promise<tf.Tensor4D> {
  let tensorsArray = await Promise.all(
    dataUrls.map(async (dataUrl) => {
      let image = new Image();
      let image_promise = onload2promise(image);
      image.src = dataUrl;
      await image_promise;
      return tf.browser.fromPixels(image, 3);
    })
  );
  let stackedTensor = tf.stack(tensorsArray) as tf.Tensor4D;
  let stackedFloatTensor = tf.cast(stackedTensor, "float32");
  tf.dispose(stackedTensor);
  tf.dispose(tensorsArray);

  return stackedFloatTensor;
}

export async function downloadNameTensorMap(namedTensorMap: NamedTensor4DMap) {
  return Promise.all(
    Object.entries(namedTensorMap).map(async ([name, tensor]) => {
      if (tensor.shape[3] === 1) {
        let newTensor = convertMaskToColors(tensor);
        await downloadTensorAsImage(newTensor, [name]);
        tf.dispose(newTensor);
        return Promise.resolve();
      }
      downloadTensorAsImage(tensor, [name]);
      return Promise.resolve();
    })
  );
}

export async function converTensorToDataUrls(
  tensor: tf.Tensor4D
): Promise<string[]> {
  let int32tensor = tf.cast(tensor, "int32");
  let depth = int32tensor.shape[-1];
  let newtensor;
  if (depth === 1) {
    newtensor = tf.tile(int32tensor, [1, 1, 1, 3]);
  } else {
    newtensor = int32tensor;
  }
  let individualTensors = tf.unstack(newtensor) as tf.Tensor3D[];
  let height = int32tensor.shape[1];
  let width = int32tensor.shape[2];
  let dataUrls = await Promise.all(
    individualTensors.map(async (individualTensor) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      await tf.browser.toPixels(individualTensor, canvas);
      return canvas.toDataURL(); // will return the base64 encoding
    })
  );
  tf.dispose(int32tensor);
  tf.dispose(newtensor);
  tf.dispose(individualTensors);
  return dataUrls;
}

/**
 * This method assumes everything in name tensor map has the same batch size;
 *
 */
export function padNamedTensorMap(
  existingNameTensorMap: NamedTensor4DMap,
  batchSize: number
): [NamedTensor4DMap[], number] {
  let currentBatchSize = Object.values(existingNameTensorMap)[0].shape[0];
  let remainder = currentBatchSize % batchSize;

  let namedTensorMapList = [];
  for (let i = 0; i < Math.ceil(currentBatchSize / batchSize); i++) {
    let namedTensorMap: NamedTensor4DMap = {};
    Object.entries(existingNameTensorMap).forEach(
      ([name, tensor]: [string, Tensor4D]) => {
        if (i === Math.ceil(currentBatchSize / batchSize) && remainder !== 0) {
          let sliced_tensor = tf.slice(
            tensor,
            i * batchSize,
            batchSize - remainder
          );
          let new_tensor = tf.pad4d(tensor, [
            [0, remainder],
            [0, 0],
            [0, 0],
            [0, 0],
          ]);
          namedTensorMap[name] = new_tensor;
          namedTensorMap[name] = tf.slice(tensor, i * batchSize, batchSize);
        }
      }
    );
    namedTensorMapList.push(namedTensorMap);
  }

  return [namedTensorMapList, remainder];
}
