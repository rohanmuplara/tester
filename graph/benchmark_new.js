async function benchmarkInput (model_path, tensors, num_runs) {
  console.time("tf.loadGraphModeling time");
  let model = await tf.loadGraphModel(model_path);
  console.timeEnd("tf.loadGraphModeling time");
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
    let grapyModel = models[0];
    let segModel = models[1];
    let tpsModel = models[2];
    let tomModel = models[3];
    let blazefaceModel = models[4];
    let densposeModel = models[5];
    let tensor1 = tf.ones([1, 256, 256,3]);
    let grapyTensor = [tf.ones([1, 512, 256, 3])];;
    let densposeTensor = [tf.ones([1, 256, 192, 3])];
    let blazefaceTensor = [tf.ones([1, 256, 256, 3])];
    let segTensor = [tensor1, tensor2, tensor3, tensor4, tensor5];

   
    benchmarkInput(models);
}

benchmarkInputDefininedInCode();
