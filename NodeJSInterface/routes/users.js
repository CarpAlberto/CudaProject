const express = require('express');
const router = express.Router();
const User = require('../models/user');
const passport = require('passport');
const jwt = require('jsonwebtoken');
const config = require('../config/database');

router.get('/', (request,response) => {
    response.send('Index Page');
})

router.post('/register', (request, response) => {
    let newUser = new User({
        name: request.body.name,
        email: request.body.email,
        username: request.body.username,
        password: request.body.password
    });

    User.addUser(newUser, (error, user) => {
        if (error) {
            response.json({ success: false, msg: 'Failed to register user' });
        } else {
            response.json({ success: true, msg: 'User registered' });
        }
    });
});

router.post('/authenticate', (request, response, next) => {
    const username = request.body.username;
    const password = request.body.password;

    User.getUserByUsername(username, (err, user) => {
        if (err)
            throw err;
        if (!user)
        {
            return response.json({ success: false, msg: 'User not found' });
        }
        console.log(user);
        User.comparePassword(password, user.password, (err,isMatch) => {
            if (err) throw err;
            if (isMatch) {
                const token = jwt.sign({data:user}, config.secret, {
                        expiresIn : 604800
                });
                response.json({
                    success: true,
                    token: 'JWT ' + token,
                    user: {
                        id: user._id,
                        name: user.name,
                        username : user.username
                    }
                });
            } else {
                return response.json({ success: false, msg: 'Wrong password' });
            }
        });

    });
});

router.get('/profile', passport.authenticate('jwt', { session: false }), (req, res, next) => {
    res.json({ user: req.user });
});

module.exports = router;