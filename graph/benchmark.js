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


async function benchmarkInput (model_path, tensors, num_runs) {
  
  console.time("model loading time");
  let model = await tf.loadGraphModel(model_path);
  console.timeEnd("model loading time");
  console.time("first prediction");
  const predictions = await runModel(model, tensors, false);
  console.timeEnd("first prediction");

  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictions = await runModel(model, tensors, true);
    let end= window.performance.now();
    let time = (end-begin) ;
    subsequent_times[i] = time;
  }
  console.log("subsequent predictions are in ms", subsequent_times);
  console.log("the average of the subequent predictions are", average(subsequent_times));
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

function benchmarkInputDefininedInCode() {
    let tensor1 = tf.ones([256, 256,3]);
    tensor1 = tensor1.expandDims(0);
    benchmarkInput("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/blazeface/tfjs/model.json.gz", [tensor1], 100);
}
benchmarkInputDefininedInCode();
