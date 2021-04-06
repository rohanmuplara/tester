async function benchmarkInput (model_path, tensors, num_runs) {
  console.time("model loading time");
  //const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/multipleoutputs3/model.json");
  console.timeEnd("model loading time");
  console.time("first prediction");
  const predictions = await runModel(model, tensors, false);
  console.timeEnd("first prediction");

  let subsequent_times =new Float32Array(num_runs - 1);
  for (let i = 0; i < num_runs - 1 ; i++) {
    let begin= window.performance.now();
    const predictions = await runModel(model, tensors, false);
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
    let tensor1 = tf.ones([5]);
    tensor1 = tensor1.expandDims(0);
    let tensor2 = tf.ones([5]);
    tensor2 = tensor2.expandDims(0);
    benchmarkInput("https://storage.googleapis.com/uplara_tfjs/multipleoutputs3/model.json", [tensor1, tensor2], 1000);
}

