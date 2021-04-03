function download_output(filename, text) {
    var element = document.createElement('a');

    element.setAttribute('href', 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(text)));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

function download_tensor_remotely(path) {
  console.log("download tensor remotely");
  let tensor = fetch('https://storage.googleapis.com/uplara_tfjs/data_file.json')
  .then(response => response.json())
  .then(data => console.log(data));

}

function testUploadedTensor(array) {
    let tensor = tf.tensor(array);
    tensor = tensor.expandDims(0);
    console.log("the tensor insider is test uploaded array is", tensor);
    benchmarkInput([tensor], 10);
}

async function compareValues() {
    console.log("compare values");
    const model = await tf.loadGraphModel("https://storage.googleapis.com/uplara_tfjs/multipleoutputs5/model.json");
    let tensor1 = tf.ones([7]);
    tensor1 = tensor1.expandDims(0);
    let tensor2 = tf.ones([8]);
    tensor2 = tensor2.expandDims(0);
    tensors = [tensor1, tensor2];
    const predictions = await runModel(model, tensors, false);
    download_output("results.txt",predictions);
}


compareValues();
