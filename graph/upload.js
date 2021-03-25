function initializeFileUploader() {
  const input = document.querySelector('input');
 input.addEventListener('input', getFile);

}

function getFile(event) {
	const input = event.target
  if ('files' in input && input.files.length > 0) {
	  placeFileContent(
      input.files[0])
  }
}

function placeFileContent(file) {
	readFileContent(file).then(content => {
    let values = JSON.parse(content);
    testUploadedTensor(values);
  }).catch(error => console.log(error))
}

function readFileContent(file) {
	const reader = new FileReader()
  return new Promise((resolve, reject) => {
    reader.onload = event => resolve(event.target.result)
    reader.onerror = error => reject(error)
    reader.readAsText(file)
  })
}


initializeFileUploader();

