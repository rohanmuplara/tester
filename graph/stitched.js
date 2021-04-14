async function benchmarkStichedInput (models, num_runs) {
    

    let model1 = await  tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet/model.json")
    let model2 = await  tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet4/model.json")
    let tensor3 = [tf.ones([1,224, 224, 3])];
    let tensor2 = [tf.ones([1,224, 224, 3])];
  for  (let i = 0; i < num_runs; i++) {
    console.time("model2  prediction" + i);
    const predictions2 = await runModel(model2, tensor3, false);
    console.timeEnd("model2  prediction" + i);
    console.time("model1  prediction" + i);
    const predictions1 = await runModel(model1, tensor2, false);
    console.timeEnd("model1  prediction" + i);
    console.timeEnd("pass" + i);
  }
}

function average(array) {
    let average = array.reduce((a, b) => a + b) / array.length;
    return average;
}

async function benchmarkStichedInputDefininedInCode() {
   console.log("tf.loadGraphModeling time");
   console.time("tf.loadGraphModeling times");
    let models = await Promise.all([
      tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet/model.json"),
      tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet2/model.json")
    ])
    console.timeEnd("tf.loadGraphModeling times");
    benchmarkStichedInput(models, 10);
}

benchmarkStichedInputDefininedInCode();
