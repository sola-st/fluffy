/**
 * Evaluate the given string.
 *
 * @param {string} x - The string to evaluate.
 */
module.exports = function (x) {
    eval(x);
};

function foo(y) {
    eval(y);
}