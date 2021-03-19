async function tester () {
  console.time("a");
  const tensor1 = tf.ones([256, 28, 28]);
  const new_tensor1 = tensor1.expandDims(0);
  const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/final5/model.json");
  console.timeEnd("a");
  const predictions =  model.execute([tensor1]);
  debugger;
  console.log("predictions");
}
tester();
