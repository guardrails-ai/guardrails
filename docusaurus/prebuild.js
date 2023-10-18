const fs = require("fs");
const path = require("path");

function processFile(relativeFilePath) {
  // Read the input file
  fs.readFile(relativeFilePath, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading file: ${err}`);
      return;
    }

    // Define a regular expression pattern to match <HTMLOutputBlock center>...</HTMLOutputBlock> tags
    const pattern = /<HTMLOutputBlock center>(.*?)<\/HTMLOutputBlock>/gs;

    let index = 0;
    // Replace matched tags with HTML fragments and save them to separate files
    data = data.replace(pattern, (_, match) => {
      // Remove whitespace and additional prefix/suffix
      let cleanedHTML = match.trim().replace(/^```html\s*/, '').replace(/\s*```$/, '');
      // remove all internal newlines and replace with a single space
      cleanedHTML = cleanedHTML.replace(/\n/g, '<br />');
      // escape all double quotes that aren't already escaped
      cleanedHTML = cleanedHTML.replace(/(?<!\\)"/g, '\\"');

      return `<CodeOutputBlock dangerouslySetInnerHTML={{ __html: "${cleanedHTML }"}} />`;
    });

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
          id: path.relative("./", path.join(dir, id)).substring('docs/'.length),
          label: snakeCaseToSentence(id),
        });
      }
    }
  });
  return fileList;
}

const examples = getFilesRecursive("./docs");

// write examples object out to file
fs.writeFileSync("./docusaurus/examples-toc.json", JSON.stringify(examples, null, 2), "utf8");


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
            console.log('Processing complete.');
          }
        });
      });
    }
  });
}

const apiReferenceDir = "./docs/api_reference_markdown/markdown";
escapeHtml(apiReferenceDir);