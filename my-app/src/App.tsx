import { useRef } from "react";
import "./App.css";
import { End_to_End_Tops } from "./run";

function App() {
  const refContainer = useRef(new End_to_End_Tops());

  return <div className="App"></div>;
}

export default App;
