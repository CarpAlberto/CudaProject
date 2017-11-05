var express = require('express');
var path = require('path')
var router = express.Router();
var ffi = require('ffi');

router.get('/', function(req, res) {
  res.render('primes', {target:type});
});


function runBasicTest() {
    var libTests = ffi.Library('../x64/Debug/TestCudaRunTime', {
        run_constructor_cpu: ['char*', []]
    });

    const pResult = libTests.run_constructor_cpu();
    try {
        return pResult.readCString(); 
    }
    finally {
        // TODO free the resources
    }
}

router.get('/basic', function(req, res) {

    let msg = runBasicTest();
    console.log(msg);

    res.send(msg);
    
});

module.exports = router;
