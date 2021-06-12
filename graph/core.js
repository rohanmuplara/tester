async function runModel(model, tensorMap, tensorOutputNames, returnTensorReferences ) {
    let num_outputs = model.outputs.length;
    let renamedTensorMap = {};
    for (const tensor_name in  tensorMap) {
      const new_tensor_name = tensor_name + ":0"
      renamedTensorMap[new_tensor_name] = tensorMap[tensor_name]
    }
    const predictionsTensor =  await model.executeAsync(renamedTensorMap);
    debugger;
    if (returnTensorReferences) {
      return constructMap(tensorOutputNames, predictionsTensor)
    } else {
      if (Array.isArray(predictionsTensor)) {
        const promises = predictionsTensor.map(x => x.array());
        const arrayTensor = await Promise.all(promises);
        tf.dispose(predictionsTensor);
        return constructMap(tensorOutputNames, arrayTensor);
      } else {
        const arrayTensor = predictionsTensor.arraySync()
        tf.dispose(predictionsTensor);
        return constructMap(tensorOutputNames, arrayTensor)
      }
    }
}
function constructMap(names, arrayValues) {
  output_dict = {}
  for (let i = 0; i < names.length; i++) {
    let name = names[i]
    let arrayValue = arrayValues[i]
    output_dict[name] = arrayValue
  }

}
async function drawPixelsToCanvas(tensor) {
  const canvas = document.createElement('canvas');
  canvas.width = tensor.shape.width
  canvas.height = tensor.shape.height
  await tf.browser.toPixels(tensor, canvas);
}
/*
This takes a raw mask and gives it colors. This is noninituive and little hacking of the api. The params is actually the color array and the mask is indicies
as each mask has a class(integer) that corresponds to the collars array. The batch size,width,height is from mask
and the depth is from the colors array.
*/

function convertMaskToColors(tensor) {
    colors = tf.tensor(
        [
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
        ]
    )
    return tf.gather(colors, mask)
}

function download(filename, text) {
    var element = document.createElement('a');

    element.setAttribute('href', 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(text)));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}


function testUploadedTensor(array) {
    let tensor = tf.tensor(array);
    tensor = tensor.expandDims(0);
    console.log("the tensor insider is test uploaded array is", tensor);
    benchmarkInput([tensor], 10);
}
