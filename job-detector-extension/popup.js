document.getElementById("check").addEventListener("click", async () => {
  try {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.body.innerText
    }, async (results) => {

      if (!results || !results[0]) {
        document.getElementById("result").innerText = "Error reading page";
        return;
      }

      let text = results[0].result;

      console.log("Extracted text:", text);

      document.getElementById("result").innerText = "Analyzing...";

      let response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
      });

      let data = await response.json();

      console.log("API Response:", data);

      document.getElementById("result").innerText = data.result;

    });

  } catch (error) {
    console.error(error);
    document.getElementById("result").innerText = "Error occurred";
  }
});