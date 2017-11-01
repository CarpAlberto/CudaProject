'use strict';
var express = require('express');
var ffi = require('ffi');
var router = express.Router();

/* GET home page. */
router.get('/', function (req, res) {
    res.render('index', { title: 'Express' });
});
router.get('/cudaProject', function (req, res) {
    res.render('index', { title: 'Express' });
});

module.exports = router;
