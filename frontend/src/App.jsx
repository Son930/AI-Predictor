import "./App.css";
import { Routes, Route } from "react-router-dom";
import Home from "../src/Pages/Home";
import Mt5Login from "./Comp/MT5login";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/mt5-login" element={<Mt5Login />} />
      <Route path="*" element={<div>404 Not Found</div>} />
    </Routes>
  );
}

export default App;
