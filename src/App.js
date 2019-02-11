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
        <h1>Illinois Tech Celebrities</h1>
        <p>Here are some you might like:</p>
        <ul>{tiles}</ul>
      </div>
    );
  }
}

export default App;
