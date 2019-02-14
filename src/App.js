import React, { Component } from 'react';
// import './App.css';


class Tile extends Component {
  render() {
    const me = this.props.celeb;
    return (
      <div className="column is-one-quarter"> 
        <div className="card">
          <div className="card-image">
            <figure className="image">
              <img src={me.image} alt= {me.name}/>
            </figure>
          </div>
          <div className="card-content">
            {me.name}
          </div>
        </div>
      </div>
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
            <div className="columns is-multiline">
              {tiles}
            </div>
        </div>
        </section>
      </div>
    );
  }
}

export default App;
