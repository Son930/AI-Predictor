import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { HeroUIProvider } from "@heroui/system";
  
import { BrowserRouter } from "react-router-dom";


import App from './App.jsx'

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <HeroUIProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </HeroUIProvider>
  </StrictMode>
);
