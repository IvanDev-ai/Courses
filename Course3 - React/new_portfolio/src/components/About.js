import React, { useEffect, useState, useRef } from "react";
import './About.css';
import Image from "./img/me.png";
import { Parallax } from 'react-scroll-parallax';

const About = () => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [startAnimation, setStartAnimation] = useState(false);
  const aboutRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      const position = window.scrollY;
      const aboutTop = aboutRef.current.getBoundingClientRect().top;

      if (aboutTop <= 0) {
        setStartAnimation(true);
      } else {
        setStartAnimation(false);
      }

      setScrollPosition(position);
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <div ref={aboutRef} className="bg-about">
      <div className="content mt-36">
        <Parallax speed={startAnimation ? -100 : 0}>
          <h2 className={`text-4xl font-bold mb-10 ${startAnimation ? 'start-animation' : ''}`} style={{ transform: `translateX(-${startAnimation ? scrollPosition / 2 : 0}px)` }}>Sobre Mí</h2>
        </Parallax>
        <p className="text-lg mb-10">Estudiante de 21 años de edad, con gran pasión por el mundo de la Inteligencia Artificial, ya sea Deep Learning, Machine Learning o NLP. Centrado en mis estudios y profesión. Eficiente y productivo.</p>
        <div className="flex flex-col justify-center items-center p-6">
          <div className="w-200 h-64 rounded-lg shadow-lg border border-gray-300 mb-4 opacity-0">
            <img src={Image} alt="Imagen 1" className="w-full h-full rounded-lg" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
