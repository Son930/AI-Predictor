import './App.css'

import { Routes, Route } from "react-router-dom";
import { createRoot } from "react-dom/client";
import Home from "../src/Pages/Home";

function App() {
  const root = document.getElementById("root");
  createRoot(root).render(<Home />);

  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        {/* Catch-all route for unknown paths */}

        <Route path="*" element={"not found"} />
      </Routes>
    </>
  );
}

export default App
