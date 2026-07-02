use anyhow::anyhow;
use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, PartialEq, TS)]
#[repr(C)]
pub enum TokenSize {
    TwoBytes,
    FourBytes,
}

impl From<TokenSize> for usize {
    fn from(value: TokenSize) -> Self {
        match value {
            TokenSize::TwoBytes => 2,
            TokenSize::FourBytes => 4,
        }
    }
}

impl TryFrom<usize> for TokenSize {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::TwoBytes),
            4 => Ok(Self::FourBytes),
            x => Err(anyhow!("Unsupported token bytes length {x}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_token_sizes_roundtrip() {
        for (size, token_size) in [(2, TokenSize::TwoBytes), (4, TokenSize::FourBytes)] {
            assert_eq!(TokenSize::try_from(size).unwrap(), token_size);
            assert_eq!(usize::from(token_size), size);
        }
    }

    #[test]
    fn unsupported_token_sizes_are_rejected() {
        for size in [0usize, 1, 3, 5, 8, usize::MAX] {
            let err = TokenSize::try_from(size).unwrap_err().to_string();
            assert_eq!(err, format!("Unsupported token bytes length {size}"));
        }
    }

    #[test]
    fn postcard_roundtrip_preserves_variant() {
        for token_size in [TokenSize::TwoBytes, TokenSize::FourBytes] {
            let decoded = psyche_test_support::postcard_roundtrip(&token_size);
            assert_eq!(decoded, token_size);
        }
    }
}
