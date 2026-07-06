use crate::Networkable;

use anyhow::Result;
use bytes::Bytes;
use iroh::{PublicKey, SecretKey};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Serialize, Deserialize)]
pub struct SignedMessage<T: Networkable> {
    from: PublicKey,
    data: Bytes,
    signature: iroh::Signature,
    _t: PhantomData<T>,
}

impl<T: Networkable> SignedMessage<T> {
    pub fn verify_and_decode(bytes: &[u8]) -> Result<(PublicKey, T)> {
        let signed_message: Self = postcard::from_bytes(bytes)?;
        let key: PublicKey = signed_message.from;
        key.verify(&signed_message.data, &signed_message.signature)?;
        let message: T = postcard::from_bytes(&signed_message.data)?;
        Ok((signed_message.from, message))
    }

    pub fn sign_and_encode(secret_key: &SecretKey, message: &T) -> Result<Bytes> {
        let data: Bytes = postcard::to_stdvec(&message)?.into();
        let signature = secret_key.sign(&data);
        let from: PublicKey = secret_key.public();
        let signed_message = Self {
            from,
            data,
            signature,
            _t: Default::default(),
        };
        let encoded = postcard::to_stdvec(&signed_message)?;
        Ok(encoded.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestMessage {
        id: u32,
        body: String,
    }

    fn key(seed: u8) -> SecretKey {
        SecretKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn signed_message_roundtrips_with_sender_key() {
        let secret = key(7);
        let message = TestMessage {
            id: 42,
            body: "hello".to_string(),
        };

        let encoded = SignedMessage::sign_and_encode(&secret, &message).unwrap();
        let (from, decoded) = SignedMessage::<TestMessage>::verify_and_decode(&encoded).unwrap();

        assert_eq!(from, secret.public());
        assert_eq!(decoded, message);
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let secret = key(8);
        let encoded = SignedMessage::sign_and_encode(
            &secret,
            &TestMessage {
                id: 1,
                body: "original".to_string(),
            },
        )
        .unwrap();
        let mut signed: SignedMessage<TestMessage> = postcard::from_bytes(&encoded).unwrap();
        signed.data = postcard::to_stdvec(&TestMessage {
            id: 1,
            body: "tampered".to_string(),
        })
        .unwrap()
        .into();
        let tampered = postcard::to_stdvec(&signed).unwrap();

        assert!(SignedMessage::<TestMessage>::verify_and_decode(&tampered).is_err());
    }

    #[test]
    fn wrong_sender_key_is_rejected() {
        let secret = key(9);
        let encoded = SignedMessage::sign_and_encode(
            &secret,
            &TestMessage {
                id: 2,
                body: "body".to_string(),
            },
        )
        .unwrap();
        let mut signed: SignedMessage<TestMessage> = postcard::from_bytes(&encoded).unwrap();
        signed.from = key(10).public();
        let tampered = postcard::to_stdvec(&signed).unwrap();

        assert!(SignedMessage::<TestMessage>::verify_and_decode(&tampered).is_err());
    }
}
