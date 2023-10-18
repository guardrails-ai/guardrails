// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");



/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Your Enterprise AI needs Guardrails",
  tagline: "Enforce assurance for LLM applications",
  favicon: "img/favicon.ico",
  // Set the production url of your site here
  url: "https://guardrailsai.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",
  staticDirectories: ['static'],
  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "Guardrails", // Usually your GitHub org/user name.
  projectName: "GuardrailsWeb", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  //mermaid
  markdown: {
    mermaid: true
  },
  // themes: ['@docusaurus/theme-mermaid', '@docusaurus/theme-classic'],
  


  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        "docs": {
          sidebarPath: require.resolve("./sidebars.js"),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
        },
      }),
    ],
  ],
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
        },
        items: [
          {
            position: "left",
            label: "Docs",
            to: "/docs",
          }
        ],
      },
      footer: {
        copyright: `Copyright © ${new Date().getFullYear()} Guardrails AI`,
      },
    }),
};

module.exports = config;
