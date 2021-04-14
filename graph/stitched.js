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

async function benchmarkStichedInput (num_runs) {
  
    
    let model1 = await  tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet/model.json")
    let model2 = await  tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/seperatemobilenet4/model.json")
    let tensor1 = [tf.ones([1,224, 224, 3])];
    let tensor2 = [tf.ones([1,224, 224, 3])];
  for  (let i = 0; i < num_runs; i++) {
    console.time("model1  prediction" + i);
    const predictions1 = await runModel(model1, tensor1, false);
    console.timeEnd("model1  prediction" + i);
     console.time("model2  prediction" + i);
    const predictions2 = await runModel(model2, tensor2, false);
    console.timeEnd("model2  prediction" + i);
  }
}


async function benchmarkStichedInputDefininedInCode() {
    console.log("bencharmking stitched input");
    benchmarkStichedInput(10);
}

benchmarkStichedInputDefininedInCode();
