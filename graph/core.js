async function runModel(model, tensors, returnTensorReferences ) {
    let num_outputs = model.outputs.length;
    let string_array = ["Identity:0"]; 
    for (let i = 1; i < num_outputs; i++) {
      string_array.push("Identity_" + i +":0");
    }
    const predictionsTensor =  await model.executeAsync(tensors, string_array);
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

