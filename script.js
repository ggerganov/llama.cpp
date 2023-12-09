#!/usr/bin/env node

'use strict';

function sum(a, b) {
	return a + b;
}

function timeout(ms, cb) {
	return new Promise(resolve => setTimeout(() => resolve(cb()), ms));
}

async function async_sum(a, b) {
	return await timeout(2000, () => sum(a, b));
}

module.exports = {
	sum,
	async_sum,
};
