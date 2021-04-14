async function benchmarkInput (models, num_runs) {
    let grapyModel = models[0];
    let segModel = models[1];
    let tpsModel = models[2];
    let tomModel = models[3];
    let blazefaceModel = models[4];
    let densposeModel = models[5];
    let tensor1 = tf.ones([1, 256, 192,3]);
    let grapyTensor = [tf.ones([1, 512, 256, 3])];
    let densposeTensor = [tf.ones([1, 256, 192, 3])];
    let blazefaceTensor = [tf.ones([1, 256, 256, 3])];
    let tomTensor = [tensor1, tensor1, tf.ones([1, 256, 192, 1])];
    let segTensor = [tf.ones([1, 256, 192, 1]), tf.ones([1, 256, 192, 27]), tensor1, tf.ones([1, 256, 192, 1]), tensor1]
    let tpsTensor = [tensor1, tf.ones([1, 256, 192, 1]), tf.ones([1, 256, 192, 1]), tensor1, tensor1];
  for  (let i = 0; i < num_runs; i++) {
  console.time("first pass");
  console.time("first grapy prediction");
  const predictionsgrapy = await runModel(grapyModel, grapyTensor, false);
  console.timeEnd("first grapy prediction");
  console.time("first seg prediction");
  const predictionsseg = await runModel(segModel, segTensor, false);
  console.timeEnd("first seg prediction");
  console.time("first tps prediction");
  const predictionstps = await runModel(tpsModel, tpsTensor, false);
  console.timeEnd("first tps prediction");
  console.time("first tom prediction");
  const predictionstom = await runModel(tomModel, tomTensor, false);
  console.timeEnd("first tom prediction");
  console.time("first blazeface prediction");
  const predictionsblazeface = await runModel(blazefaceModel, blazefaceTensor, false);
  console.timeEnd("first blazeface prediction");
  console.time("first denspose prediction");
  const predictionsdenspose = await runModel(densposeModel, densposeTensor, false);
  console.timeEnd("first denspose prediction");
  console.timeEnd("first pass");

  }


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

async function benchmarkInputDefininedInCode() {
   console.log("tf.loadGraphModeling time");
   console.time("tf.loadGraphModeling times");
    let models = await Promise.all([
      tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/grapy/atr_512_256_mobilenet_edge_loss_1/tfjs/model.json.gz"),
      tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/expseg/expected_seg_debug/tfjs/model.json.gz"),
    tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/tps/short_tshirts_default/tfjs/model.json.gz"),
    tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/tops/tom/model_60/tfjs/model.json.gz"),
    tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/blazeface/tfjs/model.json.gz"),
    tf.loadGraphModel("https://storage.googleapis.com/tfjs-alok-uplara-abcde/bottoms_gzip/densepose/densepose_js/model.json.gz")
    ])
    console.timeEnd("tf.loadGraphModeling times");
    console.log("trying to save the models");
    for (let i = 0; i < models.length; i++) {

      saved_models = await models[i].save('indexeddb://' + i);
    }
    let loadFromDb = [];
    console.time("loading from storage")
    for (let i = 0; i < models.length; i++) {

      loadFromDb.push(tf.loadGraphModel('indexeddb://' + i));
    }
    await Promise.all(loadFromDb);
    console.timeEnd("loading from storage")

    console.log("finished saving the models"); 
    benchmarkInput(models, 10);
}

benchmarkInputDefininedInCode();
