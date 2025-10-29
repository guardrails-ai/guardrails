// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

// Not actually used?
// const lightCodeTheme = require("prism-react-renderer/themes/github");
// const darkCodeTheme = require("prism-react-renderer/themes/dracula");


// TODO: Remove "new-docs" after migrating alias
const urls = {
  "production": "https://new-docs.guardrailsai.com",
  "preview": "https://preview.new-docs.guardrailsai.com",
  "": "https://docs.localhost",
};


const VERCEL_ENV = process.env.VERCEL_ENV ?? "";
const url = urls[VERCEL_ENV] ?? urls[""];


/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Your Enterprise AI needs Guardrails",
  tagline: "Enforce assurance for LLM applications",
  favicon: "img/favicon.ico",
  // Set the production url of your site here
  url: url,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/docs/", // process.env.NODE_ENV === "production" ? "/" : "/docs/",
  trailingSlash: true,
  staticDirectories: ['static'],
  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "Guardrails", // Usually your GitHub org/user name.
  projectName: "GuardrailsWeb", // Usually your repo name.

  onBrokenLinks: "warn",
  

  //mermaid
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    }
  },
  themes: ['@docusaurus/theme-mermaid'],
  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },
  plugins: [
    'docusaurus-plugin-image-zoom'
  ],
  presets: [[
    "@docusaurus/preset-classic",
    /** @type {import('@docusaurus/preset-classic').Options} */
    {
      docs: {
        routeBasePath: '/',
        sidebarPath: require.resolve("./sidebars.js"),
        sidebarCollapsed: false,
        showLastUpdateTime: true,
        
        // Please change this to your repo.
        // Remove this to remove the "edit this page" links.
        path: 'dist',
        editUrl:
          "https://github.com/guardrails-ai/guardrails/tree/main/",
      },
      blog: false,
    },
  ]],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */

    ({
      // Replace with your project's social card
      image: "img/social-card.png",
      colorMode: {
        defaultMode: "light",
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: "Guardrails AI",
        logo: {
          alt: "Guardrails Logo",
          src: "img/logo.svg",
          href: url,
          target: '_self'
        },
        items: [
          {
            type: "docSidebar",
            position: "left",
            label: "Docs",
            sidebarId: "docs",
          }, {
            type: "docSidebar",
            position: "left",
            label: "Concepts",
            sidebarId: "concepts"
          }, {
            type: "docSidebar",
            position: "left",
            label: "Tutorials",
            sidebarId: "tutorials"
          }, {
            type: "docSidebar",
            position: "left",
            label: "Integrations",
            sidebarId: "integrations"
          }, {
            type: "docSidebar",
            position: "left",
            label: "API Reference",
            sidebarId: "apiReference"
          }
        ],
      },
      footer: {
        copyright: `Copyright © ${new Date().getFullYear()} Guardrails AI`,
      },
      zoom: {
        // CSS selector to apply the plugin to, defaults to '.markdown img'
        // selector: '.markdown img',
        // Optional medium-zoom options
        // see: https://www.npmjs.com/package/medium-zoom#options
        options: {
          margin: 24,
          background: '#BADA55',
          scrollOffset: 0,
          container: '#zoom-container',
          template: '#zoom-template',
        },
      },  
    }),
};

module.exports = config;
