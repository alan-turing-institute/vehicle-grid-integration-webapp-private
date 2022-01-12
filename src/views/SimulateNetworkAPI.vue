<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-12">
        <h3>Build and simulate an electricity distribution network</h3>
        <p>
          You can find more information on OpenDSS parameters below in the
          <a
            href="http://svn.code.sf.net/p/electricdss/code/trunk/Distrib/Doc/OpenDSSManual.pdf"
            >OpenDSS Manual</a
          >.
        </p>
      </div>
    </div>

    <form id="config" ref="config" @submit.prevent>
      <div class="row box-main" style="border-color: #039BE5">
        <div class="col-lg-12 box-title" style="background-color: #B3E5FC">
          <h3>Electricity distribution network parameters</h3>
        </div>

        <div class="col-lg-6">
          <h4>Medium voltage (MV)</h4>

          <div class="form-group row">
            <!-- Experiment parameter: n_id, network ID -->
            <label for="n_id" class="col-sm-6 col-form-label">
              Network ID
            </label>
            <div class="col-sm-6">
              <select v-model="network_options.n_id" class="form-control" @change="updateLVNetworksList()">
                <option value="1060">11kV urban network</option>
                <option value="1061">11kV urban - rural network</option>
              </select>
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: xfmr_scale, MV transformer scaling -->
            <label for="xfmr_scale" class="col-md-6 col-form-label">
              MV transformer scaling
            </label>
            <div class="col-md-6">
              <input
                v-model.number="network_options.xfmr_scale"
                type="float"
                class="form-control"
                id="xfmr_scale"
                placeholder="xfmr_scale"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_setpoint -->
            <label for="oltc_setpoint" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) set point
            </label>
            <div class="col-md-6">
              <input
                v-model.number="network_options.oltc_setpoint"
                type="float"
                class="form-control"
                id="oltc_setpoint"
                placeholder="OLTC set point, e.g. 1.04"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: oltc_bandwidth -->
            <label for="oltc_bandwidth" class="col-md-6 col-form-label">
              MV transformer on-load tap charger (OLTC) bandwidth
            </label>
            <div class="col-md-6">
              <input
                v-model.number="network_options.oltc_bandwidth"
                type="float"
                class="form-control"
                id="oltc_bandwidth"
                placeholder="OLTC bandwidth, e.g. 0.13"
              />
            </div>
          </div>

          <div class="form-group row">
            <!-- Experiment parameter: rs_pen -->
            <label for="rs_pen" class="col-md-6 col-form-label">
              Proportion residential loads
            </label>
            <div class="col-md-6">
              <input
                v-model.number="network_options.rs_pen"
                type="float"
                class="form-control"
                id="rs_pen"
                placeholder="Proportion residential loads e.g. 0.8"
              />
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <h4>Low voltage (LV)</h4>

          <div class="form-group row">
            <!-- Experiment parameter: lv_default (if custom, open lv_list option below) -->
            <label for="lv_options.lv_default" class="col-md-6 col-form-label">
              Selection of LV networks to model in detail
            </label>
            <div class="col-md-6">
              <select v-model="lv_options.lv_default" class="form-control" @change="updatePreselectedLVNetworksList()">
                <option>near-sub</option>
                <option>near-edge</option>
                <option>mixed</option>
                <option>custom</option>
              </select>
            </div>
          </div>

          <div v-if="lv_options.lv_default=='custom'" class="form-group row">
            <label for="lv_list" class="col-md-6 col-form-label">
              LV network IDs
            </label>
            <div class="col-md-6">
              <select multiple class="form-control" id="lv_list" v-model="lv_options.lv_selected" :disabled="lv_options.lv_default!=='custom'">
                <option v-for="lv_id in lv_options.lv_list" :key="lv_id">{{ lv_id }}</option>
              </select>
            </div>
          </div>

          <div class="form group row">
            <label for="lv_selected_output" class="col-md-6 col-form-label">
              Currently selected LV network IDs
            </label>
            <div class="col-md-6 col-form-label">
              {{ lv_options.lv_selected.join(", ") }}
            </div>
          </div>
        </div>
      </div>

      <div class="row box-main" style="border-color: #00ACC1">
        <div class="col-lg-12 box-title" style="background-color: #B2EBF2">
          <h3>Demand and generation profiles</h3>
        </div>

        <div class="col-lg-6">
          <h4>MV connected</h4>
          <select-profile v-model:profileOptions="profile_options.mv_solar_pv" title="11kV connected solar PV profile"></select-profile>
          <select-profile v-model:profileOptions="profile_options.mv_fcs" title="11kV connected electric vehicle charging profile"></select-profile>
        </div>

        <div class="col-lg-6">
          <h4>LV connected</h4>
            <select-profile v-model:profileOptions="profile_options.lv_smart_meter" title="Smart meter"></select-profile>
            <select-profile v-model:profileOptions="profile_options.lv_electric_vehicle" title="Electric vehicles"></select-profile>
            <select-profile v-model:profileOptions="profile_options.lv_photovoltaic" title="Photovoltaic"></select-profile>
            <select-profile v-model:profileOptions="profile_options.lv_heat_pump" title="Heat pump"></select-profile>
        </div>
      </div>

      <div class="row box-main" style="border-color: #00897B">
        <div class="col-lg-12 box-title" style="background-color: #B2DFDB">
          <h3>Simulation</h3>
        </div>

        <div class="col-lg-12">
          <div class="form-group row">
            <label for="button_submit" class="col-md-3 col-form-label">
              Run simulation
            </label>
            <div class="col-md-3">
            <button type="submit" name="button_submit" class="btn btn-primary btn-block" style="background-color: #00897B; border-color: #00897B" @click="fetchAPIData">
              Submit
              <template v-if="isLoading">
                <div class="spinner-border spinner-border-sm" role="status"></div>
              </template>
            </button>
            </div>
          </div>       
          <template v-if="isLoading">
            <div class="form-group row col-md-6">
              Simulation in progress...<br>
              Usually complete in less than one minute
            </div>
          </template>
        </div>

        <div class="col-lg-12">

          <template v-if="responseAvailable">
            <h4>Results</h4>
            <div class="accordion" id="accordionResults" style="margin-bottom: 1rem">
              <div v-for="(p, ind) in plots" :key="p.ind">
                <div class="card card-results">
                  <div class="card-header">
                    <div data-toggle="collapse" :data-target="'#plotCollapse' + ind" style="float: left">
                      <i class="bi bi-chevron-down"></i>
                      {{ p.name }}
                    </div>
                    <button v-if="p.data_url !== undefined" class="btn btn-sm btn-outline-dark" type="button" style="float: right">
                        <a :href=p.data_url :download=p.data_filename>
                          <i class="bi bi-download"></i>
                          csv
                        </a>
                      </button>
                  </div>
                  <div
                    :id="'plotCollapse' + ind"
                    class="collapse"
                    v-bind:class="{ show: !ind }"
                    data-parent="#accordionResults"
                  >
                    <div class="card-body">
                      <img v-bind:src="'data:image/jpeg;base64,' + p.plot" style="max-width: 100%"/>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </template>
        </div>

      </div>
      

    </form>

    <github-link text="Website and simulation code" link="https://github.com/alan-turing-institute/vehicle-grid-integration-webapp-private"></github-link>
    <github-link text="Open profiles data" link="https://github.com/alan-turing-institute/e4Future-opendata"></github-link>
    <github-link text="Network models" link="https://github.com/alan-turing-institute/vehicle-grid-integration-opendss-networks"></github-link>

  </div>
</template>

<script>
import SelectProfile from "../components/SelectProfile.vue"
import GithubLink from "../components/GithubLink.vue"
export default {
  el: "#main",

  components: {
    SelectProfile,
    GithubLink
  },

  data() {
    return {
      network_options: {
        // Electricity distribution network parameters
        // MV
        n_id: 1060,
        xfmr_scale: 1.0,
        oltc_setpoint: 1.04,
        oltc_bandwidth: 0.013,
        rs_pen: 0.8,
      },
      lv_options:{
        lv_default: "near-sub",
        lv_list: [],
        lv_selected: []
      },
      profile_options: {
        mv_solar_pv: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
        },
        mv_fcs: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
        },
        lv_smart_meter: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
        },
        lv_electric_vehicle: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
          penetration: 1
        },
        lv_photovoltaic: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
          penetration: 1
        },
        lv_heat_pump: {
          list: [],
          profile: null,
          units: "kW",
          csv: null,
          penetration: 1
        }
      },
      // voltages: [],
      // report: [],
      rawJson: "empty",
      isShowJson: false,
      isLoading: false,
      responseAvailable: false
    };
  },

  mounted() {
    // Populate the lists used in the dropdown menus with their options
    this.getProfileOptions("mv-solar-pv").then(p_list => {
      this.profile_options.mv_solar_pv.list = p_list;
      this.profile_options.mv_solar_pv.profile = p_list[0];
    });
    this.getProfileOptions("mv-fcs").then(p_list => {
      this.profile_options.mv_fcs.list = p_list;
      this.profile_options.mv_fcs.profile = p_list[0];
    });
    this.getProfileOptions("lv-smartmeter").then(p_list => {
      this.profile_options.lv_smart_meter.list = p_list;
      this.profile_options.lv_smart_meter.profile = p_list[0];
    });
    this.getProfileOptions("lv-ev").then(p_list => {
      this.profile_options.lv_electric_vehicle.list = p_list;
      this.profile_options.lv_electric_vehicle.profile = p_list[0];
    });
    this.getProfileOptions("lv-pv").then(p_list => {
      this.profile_options.lv_photovoltaic.list = p_list;
      this.profile_options.lv_photovoltaic.profile = p_list[0];
    });
    this.getProfileOptions("lv-hp").then(p_list => {
      this.profile_options.lv_heat_pump.list = p_list;
      this.profile_options.lv_heat_pump.profile = p_list[0];
    });
    this.updateLVNetworksList()   // also updates preselected networks as nested function
  },

  methods: {

    fetchAPIData() {
      // Ask user to wait while API request is formed and made
      this.plots = [];
      this.rawJson = "wait...";
      this.isLoading = true;
      this.responseAvailable = false;

      // Initialise URL with parameters from network options (top left panel)
      var url = new URL("/simulate", process.env.VUE_APP_API_URL);
      var url_params = JSON.parse(JSON.stringify(this.network_options));
      
      // Add named set of LV networks or custom list to URL (top right panel)
      var lv_params = JSON.parse(JSON.stringify(this.lv_options));
      if (lv_params.lv_default == "custom") {
        url_params.lv_list = lv_params.lv_selected;
      } else {
        url_params.lv_default = lv_params.lv_default;
      }

      // Add MV and LV profiles to URL (middle panel, left/right for MV/LV)
      let formData = new FormData();
      url_params, formData = this.appendProfileParams(url_params, formData, "mv_solar_pv", this.profile_options.mv_solar_pv)
      url_params, formData = this.appendProfileParams(url_params, formData, "mv_fcs", this.profile_options.mv_fcs)
      url_params, formData = this.appendProfileParams(url_params, formData, "lv_smart_meter", this.profile_options.lv_smart_meter)
      url_params, formData = this.appendProfileParams(url_params, formData, "lv_ev", this.profile_options.lv_electric_vehicle)
      url_params, formData = this.appendProfileParams(url_params, formData, "lv_pv", this.profile_options.lv_photovoltaic)
      url_params, formData = this.appendProfileParams(url_params, formData, "lv_hp", this.profile_options.lv_heat_pump)

      url_params.dry_run = false;

      url.search = new URLSearchParams(url_params).toString();

      console.log("URL params", url_params)
      for (let kv of formData.entries()) {
        console.log("formData contains", kv[0], kv[1])
      }
      console.log(url)

      let check_csv_to_upload = Object.values(this.profile_options).reduce((any_csv, {profile}) => any_csv || profile=="csv", false)

      fetch(url, {
        method: "POST",
        body: check_csv_to_upload ? formData : null
      })
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          // Populate rawJson
          this.rawJson = response;

          // Parse json repsonse
          var responseJson = JSON.parse(response);

          // // Parse revoltages port from json to array to list
          // this.voltages = responseJson["voltages"];

          // // Parse report from json to array to list
          // this.report = responseJson["report"];

          // Parse plot from json to image data
          this.plots = [
            { name: "MV network overview (detailed)", plot: responseJson["mv_highlevel"] },
            { name: "MV network overview (basic)", plot: responseJson["mv_highlevel_clean"] },
            { name: "MV network powers", plot: responseJson["mv_powers"] },
            { name: "MV network voltages", plot: responseJson["mv_voltages"], data_filename: "mv_voltages.csv", data_url: URL.createObjectURL(new Blob([responseJson["mv_voltages_data"]], {type: "text/csv"})) },
            { name: "LV network voltages comparison", plot: responseJson["lv_comparison"], data_filename: "lv_comparison.csv", data_url: URL.createObjectURL(new Blob([responseJson["lv_comparison_data"]], {type: "text/csv"})) },
            { name: "Transformer powers", plot: responseJson["trn_powers"], data_filename: "transformer_powers.csv", data_url: URL.createObjectURL(new Blob([responseJson["trn_powers_data"]], {type: "text/csv"})) },
            { name: "Primary loadings", plot: responseJson["pmry_loadings"], data_filename: "primary_loadings.csv", data_url: URL.createObjectURL(new Blob([responseJson["primary_loadings_data"]], {type: "text/csv"})) },
            { name: "Primary powers", plot: responseJson["pmry_powers"] },
            { name: "Profiles", plot: responseJson["profile_options"] },
          ];

          this.responseAvailable = true;
          this.isLoading = false;
        })
        .catch(err => {
          console.log(err);
        });
    },

    appendProfileParams(url_params, form_data, name, params) {
      url_params[name + "_profile"] = params.profile;
      if (params.profile == "csv") {
        url_params[name + "_units"] = params.units;
        form_data.set(name + "_csv", params.csv[0]);
      }
      if (params.penetration !== undefined && params.profile !== "None") {
        url_params[name + "_pen"] = params.penetration;
      }
      return url_params, form_data
    },

    getProfileOptions(profile_name) {
      return fetch(
        process.env.VUE_APP_API_URL +
          "/get-options?option_type=" +
          profile_name,
        {
          method: "GET"
        }
      )
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          return JSON.parse(response);
        })
        .catch(err => {
          console.log(err);
        });
    },

    updateLVNetworksList() {
      return fetch(
        process.env.VUE_APP_API_URL + "/lv-network?n_id=" + this.network_options.n_id,
        {
          method: "GET"
        }
      )
        .then(response => {
          if (response.ok) {
            return response.text();
          } else {
            alert(
              "Server returned " + response.status + " : " + response.statusText
            );
          }
        })
        .then(response => {
          this.lv_options.lv_list = JSON.parse(response).networks;
          this.updatePreselectedLVNetworksList()
        })
        .catch(err => {
          console.log(err);
        });
    },

    updatePreselectedLVNetworksList() {
      if (this.lv_options.lv_default != "custom") {
        return fetch(
          process.env.VUE_APP_API_URL + "/lv-network-defaults?n_id=" + this.network_options.n_id + "&lv_default=" + this.lv_options.lv_default,
          {
            method: "GET"
          }
        )
          .then(response => {
            if (response.ok) {
              return response.text();
            } else {
              alert(
                "Server returned " + response.status + " : " + response.statusText
              );
            }
          })
          .then(response => {
            this.lv_options.lv_selected = JSON.parse(response).networks;
          })
          .catch(err => {
            console.log(err);
          });
      }
    }
  }
};
</script>

<style scoped>
.form-check {
  margin-bottom: 15px;
}
#displayBox {
  margin-top: 20px;
}
</style>
