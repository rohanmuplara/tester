import { useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import { Tops_Tfjs } from "./tfjs_code/TopsTfjs";

function App() {
  const refContainer = useRef<Tops_Tfjs>();
  useEffect(() => {
    // Update the document title using the browser API
    refContainer.current = new Tops_Tfjs();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    console.log("the accepted files are" + acceptedFiles);
    refContainer.current!.handle_person_upload(acceptedFiles);
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
