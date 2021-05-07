async function benchmarkStichedInput (models, num_runs, executeFirstParallely) {
    

    let model1 = models[0];
    let model2 = models[1];
    let tensor1 = [tf.ones([1,224, 224, 3])];
    let tensor2 = [tf.ones([1,224, 224, 3])];
    let first_run_index = 0;
    if (executeFirstParallely) {
      first_run_index = 1;;
      console.time("first past parallely");
      await Promise.all([
        runModel(model1, tensor2, false),
       runModel(model2, tensor1, false)
      ]);
      console.timeEnd("first past parallely");
    }
    for  (i = first_run_index; i < num_runs; i++) {

    console.time("model1  prediction" + i);
    const predictions1 = await runModel(model1, tensor2, false);
    console.timeEnd("model1  prediction" + i);

    console.time("model2  prediction" + i);
    const predictions2 = await runModel(model2, tensor1, false);
    console.timeEnd("model2  prediction" + i);
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
    benchmarkStichedInput(models, 10, false);
}

benchmarkStichedInputDefininedInCode();
