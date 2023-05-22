import child_process from 'child_process';
import fs from 'fs';

function run(cmd) { // flows to command-injection sink on next line
    child_process.execSync(cmd);
}

export { run };

export function test(
    x, // flows to code-injection sink on line 16
    y, // property `code` flows to code-injection sink on line 17
    z, // flows to low-fidelity code-injection sink on line 18
    { runme } // flows to command-injection sink on line 19
) {
    new Function(x);
    eval(y.code);
    setImmediate(z);
    child_process.spawnSync(runme);
}

export function id(x) { // flows to code-injection sink on line 26, but this flow involves a return step and hence is not flagged
    return x;
}

eval(id(42));

/**
 * This doc comment doesn't talk about the parameter we are interested in.
 *
 * @param {string} x - A parameter that doesn't exist.
 */
export default function (filename) { // flows to tainted-path sink on next line
    fs.writeFileSync(filename, '42');
}

/**
 * A doc comment for the constructor, attached to the class for some reason.
 *
 * @param {number} x - The first parameter.
 * @param {number|string} y - The second parameter.
 */
class Point {
    constructor(
        x, // doesn't flow anywhere interesting
        y // flows to code-injection sink on line 50
    ) {
        this.x = x;
        if (typeof y === 'string') {
            y = eval(y);
        }
        this.y = y;
    }
}

export { Point };