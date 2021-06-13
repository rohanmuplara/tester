import { useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import { End_to_End_Tops } from "./run";

function App() {
  const refContainer = useRef(new End_to_End_Tops());
  const onDrop = useCallback((acceptedFiles) => {
    console.log("the accepted files are" + acceptedFiles);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });
  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the files here ...</p>
      ) : (
        <p>Drag 'n' drop some files here, or click to select files</p>
      )}
    </div>
  );
}

export default App;
