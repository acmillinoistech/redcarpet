import React from 'react';
import ReactDOM from 'react-dom';
import './bulma.css';
import App from './App';
import * as serviceWorker from './serviceWorker';
import celebs from './celebs';
const CELEBS = celebs;

ReactDOM.render(<App celebs={CELEBS}/>, document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: http://bit.ly/CRA-PWA
serviceWorker.unregister();
