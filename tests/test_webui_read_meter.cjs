// sushi-test: tier=hermetic needs=node
// The prompt-read meter's text: the measured percentage, with the carried estimate beside it, through the page's own
// script under node:vm, with no browser and no model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const main = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const page = vm.createContext({});
vm.runInContext(main.slice(main.indexOf('function readEta('), main.indexOf('function createStreamingView(')), page);

// The percentage is the share of the work to do, floored: the restored prefix is not work.
assert.equal(page.readPercent(10000, 6000, 8000), 50);
assert.equal(page.readPercent(1000, 0, 996), 99);
assert.equal(page.readPercent(100, 0, 29), 29);
assert.equal(page.readPercent(100, 0, 100), 100);
assert.equal(page.readPercent(9000, 0, 0), 0);
assert.equal(page.readPercent(64000, 40000, 40000), 0);
assert.equal(page.readPercent(500, 500, 500), 100);
assert.equal(page.readPercent(100, 0, 140), 100);
assert.equal(page.readPercent(100, 60, 10), 0);
assert.equal(page.readPercent(0, 0, 0), null);

// Four formats: seconds, minutes and seconds, past an hour, no estimate yet.
const at = (total, cache, processed, time_ms) => ({ total, cache, processed, time_ms });
const text = (a) => page.readMeterText(a, a);
assert.equal(text(at(10000, 0, 400, 1000)), '4% (est. 24 s)');
assert.equal(text(at(100000, 0, 4000, 10000)), '4% (est. 4:00)');
assert.equal(text(at(1_000_000, 0, 40_000, 200_000)), '4% (est. > 1 h)');
assert.equal(text(at(1000, 0, 40, 0)), '4%');
assert.equal(text(at(1000, 0, 0, 0)), '0%');

// The text takes its percentage from the report and its estimate from the carried read: the carried read moving on
// changes the estimate and never the percentage.
const reported = at(10000, 0, 400, 1000);
assert.equal(page.readMeterText(reported, at(10000, 0, 800, 2000)), '4% (est. 23 s)');
assert.equal(page.readMeterText(reported, at(10000, 0, 9000, 3000)), '4% (est. 1 s)');
assert.equal(page.readMeterText(reported, at(10000, 0, 10000, 3000)), '4%');
