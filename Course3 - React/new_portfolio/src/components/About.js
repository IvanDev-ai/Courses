import React, { useEffect, useState, useRef } from "react";
import './About.css';
import Image from "./img/me.png";
import { Parallax } from 'react-scroll-parallax';

const About = () => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [startAnimation, setStartAnimation] = useState(false);
  const [finalTransforms, setFinalTransforms] = useState(null);
  const [showImage, setShowImage] = useState(false);

  const aboutRef = useRef(null);

  const TRANSLATE_X_LIMIT_h2 = 250; // Límite para el translateX en píxeles
  const TRANSLATE_Y_LIMIT_h2 = 10; // Límite para el translateY en píxeles
  const SCALE_LIMIT_h2 = 2; // Límite para el escalado

  const TRANSLATE_X_LIMIT_p = 250; // Límite para el translateX en píxeles
  const TRANSLATE_Y_LIMIT_p = 1; // Límite para el translateY en píxeles
  const SCALE_LIMIT_p = 1.2; // Límite para el escalado

  const TRANSLATE_X_LIMIT_img = 100; // Límite para el translateX en píxeles
  const TRANSLATE_Y_LIMIT_img = 200; // Límite para el translateY en píxeles
  const SCALE_LIMIT_img = 1.1; // Límite para el escalado
  useEffect(() => {
    const handleScroll = () => {
      const position = window.scrollY;
      const aboutTop = aboutRef.current.getBoundingClientRect().top;

      if (aboutTop <= 0 && !startAnimation) {
        setStartAnimation(true);
        setShowImage(true);
        setFinalTransforms({
          translateX_h2,
          translateY_h2,
          scale_h2,
          translateX_p,
          translateY_p,
          scale_p,
          translateX_img,
          translateY_img,
          scale_img
        });
      } else if (aboutTop > 0 && startAnimation) {
        setStartAnimation(false);
        setFinalTransforms(null);
        setShowImage(false);
      }

      setScrollPosition(position);
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [startAnimation]);

  const translateX_h2 = Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_h2);
  const translateY_h2 = Math.min(scrollPosition / 4, TRANSLATE_Y_LIMIT_h2);
  const scale_h2 = 1 + (Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_h2) / TRANSLATE_X_LIMIT_h2) * (SCALE_LIMIT_h2 - 1);

  const translateX_p = Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_p);
  const translateY_p = Math.min(scrollPosition / 4, TRANSLATE_Y_LIMIT_p);
  const scale_p = 1 + (Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_p) / TRANSLATE_X_LIMIT_p) * (SCALE_LIMIT_p - 1);

  const translateX_img = Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_img);
  const translateY_img = Math.min(scrollPosition / 4, TRANSLATE_Y_LIMIT_img);
  const scale_img = 1 + (Math.min(scrollPosition / 2, TRANSLATE_X_LIMIT_img) / TRANSLATE_X_LIMIT_img) * (SCALE_LIMIT_img - 1);

  const transformH2 = startAnimation
    ? `translate(-${finalTransforms.translateX_h2}px, -${finalTransforms.translateY_h2}px) scale(${finalTransforms.scale_h2})`
    : `translate(-${translateX_h2}px, -${translateY_h2}px) scale(${scale_h2})`;

  const transformP = startAnimation
    ? `translate(-${finalTransforms.translateX_p}px, -${finalTransforms.translateY_p}px) scale(${finalTransforms.scale_p})`
    : `translate(-${translateX_p}px, -${translateY_p}px) scale(${scale_p})`;

  const transformIMG = startAnimation
    ? `translate(-${finalTransforms.translateX_img}px, -${finalTransforms.translateY_img}px) scale(${finalTransforms.scale_img})`
    : `translate(-${translateX_img}px, -${translateY_img}px) scale(${scale_img})`;  

  const imageStyle = {
    opacity: showImage ? 1 : 1,
    transform: transformIMG,
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    transition: 'opacity 1s, box-shadow 0.5s ease-out'
  };

  return (
    <div ref={aboutRef} className="bg-about">
      <div className="content mt-32">
        <h2
          className="text-4xl font-bold mb-10"
          style={{
            transform: transformH2,
            transition: 'transform 0.1s ease-out'
          }}
        >
          Sobre Mí
        </h2>
          <p
            className="text-lg mr-4"
            style={{
              transform: transformP,
              width:startAnimation ? "60%": "90%",
              transition: 'transform 0.1s ease-out'
            }}
          >
            Estudiante de 21 años de edad, con gran pasión por el mundo de la Inteligencia Artificial, ya sea Deep Learning, Machine Learning o NLP. Centrado en mis estudios y profesión. Eficiente y productivo.
          </p>
          <div
            className="w-400 h-64 rounded-lg shadow-lg border border-gray-300"
            style={imageStyle}
          >
            <img src={Image} alt="Imagen 1" className="w-200 h-full rounded-lg" />
          </div>
      </div>
    </div>
  );
};

export default About;
