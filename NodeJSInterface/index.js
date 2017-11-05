const express = require('express');
const path    = require('path')
const bodyParser = require('body-parser');
const cors = require('cors');
const mongoose = require('mongoose');
const config = require('./config/database');
const passport = require('passport');
const app = express();

const users = require('./routes/users');
const tests = require('./routes/tests');
const PORT = 3000;

mongoose.connect(config.database);

mongoose.connection.on('connected', () => {
    console.log('connected to database');
});

app.use(express.static(path.join(__dirname, 'public')));

// Body Parser Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


// Cors middleware
app.use(cors());

// Passport middleware
app.use(passport.initialize());
app.use(passport.session());
require('./config/passport')(passport);


app.use('/users', users);
app.use('/tests', tests);

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

var server = app.listen(PORT, () => {
    console.log('Web server listing at http://localhost:%s', PORT);
});
