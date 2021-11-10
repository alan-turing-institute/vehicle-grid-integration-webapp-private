import { createRouter, createWebHistory } from "vue-router";
import NavBar from "@/layouts/NavBar.vue";

const routes = [
  {
    path: "/",
    component: NavBar,
    children: [
      {
        path: "",
        name: "Home",
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () =>
          import(/* webpackChunkName: "home" */ "../views/Home.vue")
      },
      {
        path: "analysis",
        name: "Analysis",
        component: () =>
          import(/* webpackChunkName: "analysis" */ "../views/Analysis.vue")
      },
      {
        path: "analysis/simulate_network",
        name: "SimulateNetwork",
        component: () =>
          import(
            /* webpackChunkName: "simulate_network" */ "../views/SimulateNetwork.vue"
          )
      },
      {
        path: "analysis/simulate_network_API",
        name: "SimulateNetworkAPI",
        component: () =>
          import(
            /* webpackChunkName: "simulate_network_api" */ "../views/SimulateNetworkAPI.vue"
          )
      }
    ]
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
