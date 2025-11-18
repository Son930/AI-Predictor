// main.tsx or main.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { HeroUIProvider } from "@heroui/react";
import App from "./App";
import "./index.css";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <BrowserRouter>
    <HeroUIProvider>
      <main className="dark text-foreground bg-background ">
        <App />
      </main>
    </HeroUIProvider>
    </BrowserRouter>
  </React.StrictMode>
);
