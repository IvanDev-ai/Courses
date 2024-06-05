import React, { useEffect, useRef, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import './About.css';

const About = () => {
  const aboutRef = useRef(null);
  const controls = useAnimation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const aboutSection = aboutRef.current;
      if (!aboutSection) return;

      const { top, height } = aboutSection.getBoundingClientRect();
      const windowHeight = window.innerHeight;

      if (top <= windowHeight * 0.75 && top + height >= 0) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    if (scrolled) {
      controls.start({
        x: -aboutRef.current.offsetWidth / 2 + 50, // Adjust to position as needed
        y: -aboutRef.current.offsetHeight / 2 + 50, // Adjust to position as needed
        transition: { duration: 0.5 },
      });
    } else {
      controls.start({
        x: 0,
        y: 0,
        transition: { duration: 0.5 },
      });
    }
  }, [scrolled, controls]);

  return (
    <div ref={aboutRef} className="about-section min-h-screen flex items-center justify-center bg-white text-black">
      <div className="content p-8 max-w-2xl relative">
        <motion.h2
          className="title text-4xl font-bold mb-4"
          animate={controls}
        >
          Sobre Mí
        </motion.h2>
        <p className="text-lg">
          Este apartado es dedicado a información sobre mí.
        </p>
      </div>
    </div>
  );
};

export default About;
