async function tester () {
  console.time("a");
  const tensor1 = tf.ones([256, 192,1]);
  const new_tensor1 = tensor1.expandDims(0);

  const tensor2 = tf.ones([1, 256, 192,3]);
  const tensor3 = tf.ones([1,256, 192,1]);
  const tensor4 = tf.ones([1,256, 192,3]);
  const tensor5 = tf.ones([1,256, 192,1]);
  const tensor6 = tf.ones([1, 256, 192,1]);
  const model = await tf.loadGraphModel("https://rohan-models.s3-us-west-2.amazonaws.com/model.json");
  console.timeEnd("a");
  const predictions =  model.executeAsync([new_tensor1, tensor2, tensor3, tensor4, tensor5, tensor6]);

}
tester();
