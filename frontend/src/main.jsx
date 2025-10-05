import React from 'react'; import { createRoot } from 'react-dom/client';
import App from './App'; import './style.css';
const el = document.getElementById('root') || (()=>{const n=document.createElement('div');n.id='root';document.body.appendChild(n);return n})(); 
createRoot(el).render(<App/>);