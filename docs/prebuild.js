const fs = require("fs");
const path = require("path");
const oldTabRegex = /^=== \"(.*)\"$/;


function processFile(relativeFilePath) {
  // Read the input file
  fs.readFile(relativeFilePath, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading file: ${err}`);
      return;
    }

    // Define a regular expression pattern to match <HTMLOutputBlock center>...</HTMLOutputBlock> tags
    const pattern = /<HTMLOutputBlock center>(.*?)<\/HTMLOutputBlock>/gs;
    let hasCodeBlocks = false;
    // Replace matched tags with HTML fragments and set them dangerously
    data = data.replace(pattern, (_, match) => {
      hasCodeBlocks = true;
      
      let cleanedHTML = match.trim().replace(/^```html\s*/, '').replace(/\s*```$/, '');
      return writeSafeHtml(cleanedHTML);
    });

    if (data.includes("CodeOutputBlock")) {
      hasCodeBlocks = true;
    }

    // look for an html insertion
    if (data.split('\n').find(s => s === "--8<--")) {
      hasCodeBlocks = true;
      data = extractHtml(data, relativeFilePath);
    }

    // look for an old tab match anywhere in the file
    if (data.split("\n").find(s => s.match(oldTabRegex))) {
      data = tabs(data);
    }


    // compute path to the code-output-block.jsx file
    const codeOutputBlockPath = path.relative(path.dirname(relativeFilePath), './code-output-block.jsx');
    const importStatement = `import CodeOutputBlock from '${codeOutputBlockPath}';\n\n`;
    if (hasCodeBlocks && !data.includes(importStatement)) {
      // import the code-output-block component at the top of each file
      data =  importStatement + data;
    }

    // Write the modified content back to the input file
    fs.writeFile(relativeFilePath, data, 'utf8', (err) => {
      if (err) {
        console.error(`Error writing file: ${err}`);
      } else {
        console.log('Processing complete.');
      }
    });
  });
}

function writeSafeHtml(cleanedHTML) {
  // Remove whitespace and additional prefix/suffix
  // remove all internal newlines and replace with a single space
  cleanedHTML = cleanedHTML.replace(/\n/g, '<br />');
  // escape all double quotes that aren't already escaped
  cleanedHTML = cleanedHTML.replace(/(?<!\\)"/g, '\\"');

  // remove straggling ```html lines
  cleanedHTML = cleanedHTML.replace(/```html/g, '');

  return `<CodeOutputBlock dangerouslySetInnerHTML={{ __html: "${cleanedHTML}"}} />`;
}

function extractHtml(data, relativeFilePath) {
  const splitData = data.split("\n");
  let inHtml = false;
  let htmlFilePath = '';
  for (let i = 0; i < splitData.length; i++) {
    const line = splitData[i];
    if (inHtml) {
      if (line === "--8<--") {
        inHtml = false;
        // read in html file path 
        const htmlData = fs.readFileSync(htmlFilePath, 'utf8');
        console.log(htmlData);
        const safeHtml = writeSafeHtml(htmlData);
        splitData.splice(i, 1, safeHtml); 
          
      } else if (line !== '' && line.trim() !== '') {
        htmlFilePath = "./" + line.trim();
        // console.log("\n\n\n\n\n\n\n\n" + htmlFilePath)
        splitData.splice(i, 1);
        i--;
      }
    } else if (line === "--8<--") {
      inHtml = true;
      splitData.splice(i, 1);
      i--;
    }
  }

  return splitData.join("\n");
}

function tabs(data) {
  data = `import Tabs from '@theme/Tabs';import TabItem from '@theme/TabItem';\n\n` + data;
  const splitData = data.split("\n");
  // regex for old tab style

  let inTab = false;
  let inTabGroup = false
  for (let i = 0; i < splitData.length; i++) {
    const line = splitData[i];
    if (inTab) {
      // check if the line is empty or starts with four spaces
      if (!line.startsWith("    ") && line.length !== 0 && !line.match(/^\s*$/)) {
        // we are no longer in a tab
        inTab = false;
        
        // if the line is not a tab, we're no longer in a tab group
        if (!line.match(oldTabRegex)) {
          splitData.splice(i, 0, "</Tabs>");
          inTabGroup = false;
        }

        splitData.splice(i, 0, "</TabItem>");
        i++;
      }
    }
    if (line.match(oldTabRegex)) {
      // we are in a tab
      inTab = true;
      const tabName = line.match(oldTabRegex)[1];
      if (!inTabGroup) {
        splitData.splice(i, 1, `<Tabs>`);
        splitData.splice(i+1, 0, `<TabItem value="${tabName.split(" ").join("")}" label="${tabName}">`);
        i+=2;
        inTabGroup = true;
      } else {
        splitData.splice(i, 1, `<TabItem value="${tabName.split(" ").join("")}" label="${tabName}">`);
        i++;
      }
    }
  }

  return splitData.join("\n");
}

function snakeCaseToSentence(str) {
  return str.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

function getFilesRecursive(dir) {
  const files = fs.readdirSync(dir, { withFileTypes: true });
  let fileList = [];
  files.forEach((file) => {
    const filePath = path.join(dir, file.name);

    if (file.isDirectory()) {
      const subFileItems = getFilesRecursive(filePath);
      if (subFileItems && subFileItems.length > 0) {
        fileList.push({
          type: "category",
          label: snakeCaseToSentence(file.name),
          items: subFileItems,
          collapsed: true,
          collapsible: true,
        });
      }

    } else {
      if (file.name.endsWith(".mdx") || file.name.endsWith(".md")) {
        processFile(filePath);
        const id = file.name.replace(".mdx", "").replace(".md", "");
        fileList.push({
          type: "doc",
          id: path.relative("./", path.join(dir, id)).substring('dist/'.length),
          label: snakeCaseToSentence(id),
        });
      }
    }
  });
  return fileList;
}

const examples = getFilesRecursive("./dist");

// write examples object out to file
fs.writeFileSync("./examples-toc.json", JSON.stringify(examples, null, 2), "utf8");


// escape all < and > tags in files in docs/api_reference_markdown/markdown
function escapeHtml (fp) {
  const files = fs.readdirSync(fp, { withFileTypes: true });
  files.forEach((file) => {
    const filePath = path.join(fp, file.name);

    if (file.isFile() && (file.name.endsWith(".mdx") || file.name.endsWith(".md"))) {
      fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
          console.error(`Error reading file: ${err}`);
          return;
        }

        // remove all > characters and the whitespace between them and the next character at the beginning of each line
        // regex for only spaces between characters, not newlines

        data = data.replace(/^>* */gm, '');
        data = data.replace(/^>/gm, '');
        data = data.replace(/</g, "&lt;");

        fs.writeFile(filePath, data, 'utf8', (err) => {
          if (err) {
            console.error(`Error writing file: ${err}`);
          } else {
            console.log(`Successfully processed file ${filePath}.`);
          }
        });
      });
    }
  });
}

const apiReferenceDir = "./dist/api_reference_markdown";
// escapeHtml(apiReferenceDir);