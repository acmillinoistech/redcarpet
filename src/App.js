import React, { Component } from 'react';
// import './App.css';


class Tile extends Component {
  render() {
    const me = this.props.celeb;
    return (
      <li>{me.name}</li>
    );
  }
}


class App extends Component {
  render() {
    const tiles = this.props.celebs.map((celeb, idx) =>
      <Tile key={idx} celeb={celeb} />
    );
    console.log(this.props.tiles)
    return (
      <div className="App">
        <section className="hero is-info is-bold">
          <div className="hero-body">
            <div className="container">
              <h1 className="title">Illinois Tech Celebrities</h1>
            </div>
          </div>
        </section>
        <section className="section">        
        <div className="container content">
            <p>Here are some you might like:</p>
            <ul>{tiles}</ul>
        </div>
        </section>
      </div>
    );
  }
}

export default App;
