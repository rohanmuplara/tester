import * as tf from "@tensorflow/tfjs";

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

export async function drawPixelsToCanvas(tensor: tf.Tensor, name: string) {
  const canvas = document.createElement("canvas");
  canvas.width = tensor.shape[0];
  canvas.height = tensor.shape[1]!;
  let squeezed_tensor: tf.Tensor3D = tf.squeeze(tensor, [0]);
  let squeezed_int_tensor = tf.cast(squeezed_tensor, "int32");
  await tf.browser.toPixels(squeezed_int_tensor, canvas)!;
  let fake_link = document.createElement("a");
  fake_link.download = name;
  fake_link.href = canvas.toDataURL();
  fake_link.click();
}
/*
This takes a raw mask and gives it colors. This is noninituive and little hacking of the api. The params is actually the color array and the mask is indicies
as each mask has a class(integer) that corresponds to the collars array. The batch size,width,height is from mask
and the depth is from the colors array.
*/

export function convertMaskToColors(mask: tf.Tensor) {
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
  return tf.gather(colors, spliced_mask);
}

async function downloadImage(image_url: string) {
  let image = new Image();
  image.crossOrigin = "anonymous";
  let image_promise = onload2promise(image);
  image.src = image_url;
  await image_promise;
  return image;
}

export async function convertMaskUrlToTensor(mask_url: string) {
  let mask_image = await downloadImage(mask_url);
  return tf.browser.fromPixels(mask_image, 1);
}

export async function convertImageUrlToTensor(image_url: string) {
  let image = await downloadImage(image_url);
  return tf.browser.fromPixels(image, 3);
}

export async function handle_image_load(files: [any]) {
  console.log("in handle image load");
  let image = new Image();
  let fr = new FileReader();

  fr.onload = function () {
    image.src = fr.result as string;
  };
  let image_promise = onload2promise(image);
  fr.readAsDataURL(files[0]);
  console.log("awaiting image promise");
  await image_promise;
  let result = tf.browser.fromPixels(image, 3);

  return result;
}

interface OnLoadAble {
  onload: any;
}
function onload2promise<T extends OnLoadAble>(obj: T): Promise<T> {
  return new Promise((resolve, reject) => {
    obj.onload = () => resolve(obj);
  });
}
