var casper = require("casper").create({
    viewportSize: {
        width: 1366,
        height: 650
    }
});
var fs = require("fs");

casper.options.pageSettings = {
    loadImages:  true, 
    loadPlugins: true,
    userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1588.0 Safari/537.36"
};

function getCategories() {
    var span = document.querySelector("h3").nextSibling;
    var searching = true;
    var categorySpans = [];
    while (searching) {
        if (!span.nextSibling) {
            break;
        }
        categorySpans.push(span.href);
        span = span.nextSibling;
    }
    return categorySpans;
}

function getTalent() {
    var celebs = [];
    var els = document.querySelector('#talent-grid').children;
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        celebs.push({
            name: el.querySelector('h4').innerText,
            price: parseInt(el.querySelector('span').innerText.split("\n")[0].split("$")[1]),
            url: el.href
            // "tags": el.querySelector('span').innerText.trim().split("\n- ")[1].split(" - ")
            // "image": el.querySelector('img').getAttribute("src")
        });
    }
    return celebs;
}

function getTalentDetails() {
    var arr = document.querySelectorAll("a");
    var categories = [];
    for (var i = 0; i < arr.length; i++) {
        if (arr[i].id.indexOf("user-category-button") > -1) {
            categories.push(arr[i].innerText.trim());
        }
    }
    var h4s = document.querySelectorAll("h4");
    var ratingComps = "";
    if (h4s.length >= 4) {
        ratingComps = h4s[3].innerText;
    }
    var reactions = 0;
    var stars = 0;
    if (ratingComps.indexOf(" | ") > -1) {
        ratingComps = ratingComps.split(" | ");
        reactions = parseInt(ratingComps[0].split(" reactions"));
        stars = parseFloat(ratingComps[1].split(" stars"));
    }
    // var responds = document.querySelector("[aria-label='fire emoji']").parentElement.innerText.slice(2);
    var joined = document.querySelector(".aD5HqrCP8tV5nVRjIoayM").innerText.trim();
    return {
        categories: categories,
        reactions: reactions,
        stars: stars,
        // responds: responds,
        joined: joined
    };
}


casper.start("https://www.cameo.com/c/featured");

casper.then(function() {
    var urls = this.evaluate(getCategories);
    // urls = urls.slice(0, 3);
    // for (var i in urls) {
    //     this.echo("Found: " + urls[i]);
    // }
    var data = [];
    var total = 0;
    var curr = 0;
    this.eachThen(urls, function(item) {
        var url = item.data;
        this.thenOpen(url, function(response) {
            var celebs = this.evaluate(getTalent);
            // celebs = celebs.slice(116);
            curr++;
            var idx = 0;
            this.echo("Opened (" + curr + "/" + urls.length + "): " + url.split("/c")[1] + " (" + celebs.length + ") pages.");
            this.eachThen(celebs, function(item) {
                var celeb = item.data;
                this.thenOpen(celeb.url, function(response) {
                    total++;
                    idx++;
                    this.echo("Visited (" + idx + "/" + celebs.length + "): " + celeb.url);
                    // this.captureSelector("err/" + celeb.name + ".png", "body");
                    var details = this.evaluate(getTalentDetails);
                    var talent = {
                        name: celeb.name,
                        price: celeb.price,
                        url: celeb.url,
                        categories: details.categories,
                        reactions: details.reactions,
                        stars: details.stars,
                        // responds: details.responds,
                        joined: details.joined
                    };
                    data.push(talent);
                });
            });
        });
    });
    this.then(function() {
        this.echo("Visited " + total + " pages.");
        this.echo("Wrote " + data.length + " records to file.");
        fs.write("output.json", JSON.stringify(data), "w");
    });
});

casper.run();