async function runModel(model, tensors, returnTensorReferences ) {
    let num_outputs = model.outputs.length;
    const predictionsTensor =  await model.executeAsync(tensors, ["module_apply_default/hub_output/feature_vector/SpatialSqueeze"]);
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
  let model = await tf.loadGraphModel(model_path, { fromTFHub: true });
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
    let tensor1 = tf.ones([224, 224,3]);
    tensor1 = tensor1.expandDims(0);
    benchmarkInput("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/2/default/1", [tensor1], 100);
}
benchmarkInputDefininedInCode();
