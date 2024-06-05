import React, {useEffect} from 'react';
import './App.css';
import Portada from './components/portada';
import About from './components/About';
import Contact from './components/Contact'
import Resume from './components/Resume';
import Portfolio from './components/Portfolio'
import {Link, Element} from 'react-scroll';

function App() {
  useEffect(() =>{
    const handleScroll = () => {
      const header = document.querySelector('header');
      if (window.scrollY >= (window.innerHeight - 500)){
        header.classList.add('visible');
      } else {
        header.classList.remove('visible');
      };
    }
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    }
  },[]);

  return (
    <div className='App'>
      <header className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-2/5 bg-glass dark:bg-glass-dark rounded-lg shadow-md z-10">
        <nav className="container mx-auto p-4 flex justify-center space-x-4">
          <Link to="portada" smooth={true} duration={200} className="cursor-pointer">
            Portada
          </Link>
          <Link to="about" smooth={true} duration={200} className="cursor-pointer">
            Sobre Mí
          </Link>
          <Link to="portfolio" smooth={true} duration={200} className='cursor-pointer'>
            Portfolio
          </Link>
          <Link to="resume" smooth={true} duration={200} className='cursor-pointer'>
          Resumen
          </Link>
          <Link to="contact" smooth={true} duration={200} className='cursor-pointer'>
            Contacto
          </Link>
        </nav>
      </header>
      <Element name="portada">
        <Portada />
      </Element>
      <Element name="about">
        <About />
      </Element>
      <Element name='portfolio'>
        <Portfolio/>
      </Element>
      <Element name="resume">
        <Resume />
      </Element>
      <Element name='contact'>
        <Contact/>
      </Element>
    </div>
  );
}

export default App;
