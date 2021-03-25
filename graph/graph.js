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
        const numpyResults = await Promise.all(promises);
      } else {
        const predictionsTensor = predictionsTensor.arraySync()
      }
    }
}



async function benchmarkInput (tensors, num_runs) {
  console.time("model loading time");
  const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/multipleoutputs3/model.json");
  console.timeEnd("model loading time");
  console.time("first prediction");
  const predictions = await runModel(model, tensors, false);
  console.log("the predictions are", predictions);
  download("results.json", predictions);
  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictionsTensor =  await model.executeAsync(tensors);
    let end= window.performance.now();
    let time = (end-begin) ;
    subsequent_times[i] = time;
  }
  console.log("subsequent predictions are in ms", subsequent_times);
  console.log("the average of the subequent predictions are", average(subsequent_times));
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

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

function benchmarkInputDefininedInCode() {
    let tensor1 = tf.ones([5]);
    tensor1 = tensor1.expandDims(0);
    let tensor2 = tf.ones([5]);
    tensor2 = tensor2.expandDims(0);
    benchmarkInput([tensor1, tensor2], 10);
}

function testUploadedTensor(array) {
    let tensor = tf.tensor(array);
    tensor = tensor.expandDims(0);
    console.log("the tensor insider is test uploaded array is", tensor);
    benchmarkInput([tensor], 10);
}

benchmarkInputDefininedInCode();
