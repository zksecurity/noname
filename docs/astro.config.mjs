// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  integrations: [
    starlight({
      title: "noname docs",
      social: {
        github: "https://github.com/zksecurity/noname",
      },
      sidebar: [
        {
          label: "Guides",
          //items: [{ label: "Hello World", slug: "guides/hello_world" }],
          autogenerate: { directory: "guides" },
        },
        {
          label: "Reference",
          autogenerate: { directory: "reference" },
        },
      ],
    }),
  ],
});
