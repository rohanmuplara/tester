async function runModel(model, tensorMap, returnTensorReferences ) {
    let num_outputs = model.outputs.length;
    let renamedTensorMap = {};
    for (const tensor_name in  tensorMap) {
      const new_tensor_name = tensor_name + ":0"
      renamedTensorMap[new_tensor_name] = tensorMap[tensor_name]
    }
    const predictionsTensor =  await model.executeAsync(renamedTensorMap);
    if (returnTensorReferences) {
      return predictionsTensor;
    } else {
      if (Array.isArray(predictionsTensor)) {
        const promises = predictionsTensor.map(x => x.array());
        const arrayTensor = await Promise.all(promises);
        tf.dispose(predictionsTensor);
        return arrayTensor;
      } else {
        const arrayTensor = predictionsTensor.arraySync()
        tf.dispose(predictionsTensor);
        return arrayTensor;
      }
    }
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

