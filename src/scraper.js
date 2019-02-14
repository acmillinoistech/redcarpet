const els = Array.from(document.querySelector('#talent-grid').children);
const celebs = els.map((el) => {
    return {
        "name": el.querySelector('h4').innerText,
        "price": parseInt(el.querySelector('span').innerText.split("\n")[0].split("$")[1]),
        "tags": el.querySelector('span').innerText.trim().split("\n- ")[1].split(" - "),
        "image": el.querySelector('img').getAttribute("src")
    }
});
document.write(JSON.stringify(celebs));