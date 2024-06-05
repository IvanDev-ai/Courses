import React from "react";
import './portada.css';
import Icon from './img/icon.png';
const Portada = () => {
    const textArray = Array(5).fill("IVÁN L. MARTÍN")
    return (
        <div className="relative min-h-screen flex flex-col items-center justify-center bg-black text-white overflow-hidden">
            <div className="z-10 text-center">
                <h1 className="text-6xl mb-10">Bienvenido a mi Portfolio</h1>
                <img src={Icon} alt="Portfolio Image" className="w-2/5 max-w-lg mx-auto mb-8"/>
                <div className="bottom-4 text-center w-full">
                    <p className="mb-2">Desliza hacia abajo</p>
                    <svg className="w-6 h-6 mx-auto animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                    </svg>
                </div>
            </div>
            <div className="absolute top-0 w-full h-full flex items-center justify-center z-0">
                {
                    textArray.map((text, index) => (
                        <div key={index} className="text-9xl font-bold px-5 text-white opacity-30 animate-scroll repeat-text whitespace-nowrap">{text}</div>
                    ))
                }
            </div>
        </div>
    )
}
export default Portada;