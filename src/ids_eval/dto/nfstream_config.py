from pydantic import BaseModel, Field


class NFStreamConfig(BaseModel):
    idle_timeout:           int = Field(default = 120, description="idle_timeout")
    active_timeout:         int = Field(default=1800, description="active_timeout")
    statistical_analysis:   bool= Field(default=True, description="statistical_analysis")
    splt_analysis:          int = Field(default=0, description="splt_analysis")
    n_dissections:          int = Field(default=0, description="n_dissections")
    accounting_mode :       int = Field(default=0, description="accounting_mode")
    n_meters:               int = Field(default=0, description="n_meters")
    performance_report:     int = Field(default=0, description="performance_report")
    decode_tunnels:         bool= Field(default= False, description="decode_tunnel")
    bpf_filter:             str = Field(default="ip", description="bpf_filter")
    snapshot_length:        int = Field(default=1536, description="snapshot_length")
    system_visibility_mode: int = Field(default=0, description="system_visibility_mode")
